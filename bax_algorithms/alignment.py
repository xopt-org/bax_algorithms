class ScipyBeamAlignment(Algorithm, ABC):
    name = "ScipyBeamAlignment"
    meas_dims: Union[int, list[int]] = Field(
        description="list of indeces identifying the measurement quad dimensions in the model"
    )
    x_key: str = Field(None,
        description="oberservable name for x centroid position"
    )
    y_key: str = Field(None,
        description="oberservable name for y centroid position"
    )

    @property
    def observable_names_ordered(self) -> list:  
        # get observable model names in the order they appear in the model (ModelList)
        return [key for key in [self.x_key, self.y_key] if key]
    
    def get_execution_paths(
        self, model: ModelList, bounds: Tensor, verbose=False
    ) -> Tuple[Tensor, Tensor, Dict]:
        """get execution paths that minimize the objective function"""

        meas_scans = torch.index_select(
            bounds.T, dim=0, index=torch.tensor(self.meas_dims)
        )
        ndim = bounds.shape[1]
        tuning_dims = [i for i in range(ndim) if i not in self.meas_dims]
        tuning_domain = torch.index_select(
            bounds.T, dim=0, index=torch.tensor(tuning_dims)
        )

        device = torch.tensor(1).device
        torch.set_default_tensor_type("torch.DoubleTensor")

        cpu_models = [copy.deepcopy(m).cpu() for m in model.models]
        if MaternKernel in [type(k) for k in cpu_models[0].covar_module.base_kernel.kernels]:
            sample_funcs_list = [
                draw_product_kernel_post_paths(cpu_model, n_samples=self.n_samples)
                    for cpu_model in cpu_models
                ]
        else:
            sample_funcs_list = [
                draw_linear_product_kernel_post_paths(cpu_model, n_samples=self.n_samples)
                    for cpu_model in cpu_models
                ]

        xs_tuning_init = unif_random_sample_domain(
            self.n_samples, tuning_domain
        ).double()

        x_tuning_init = xs_tuning_init.flatten()

        # minimize
        def target_func_for_scipy(x_tuning_flat):
            return (
                self.sum_samplewise_misalignment_flat_x(
                    sample_funcs_list,
                    torch.tensor(x_tuning_flat),
                    self.meas_dims,
                    meas_scans.cpu(),
                )
                .detach()
                .cpu()
                .numpy()
            )

        def target_func_for_torch(x_tuning_flat):
            return self.sum_samplewise_misalignment_flat_x(
                sample_funcs_list, x_tuning_flat, self.meas_dims, meas_scans.cpu()
            )

        def target_jac(x):
            return (
                torch.autograd.functional.jacobian(
                    target_func_for_torch, torch.tensor(x)
                )
                .detach()
                .cpu()
                .numpy()
            )

        res = minimize(
            target_func_for_scipy,
            x_tuning_init.detach().cpu().numpy(),
            jac=target_jac,
            bounds=tuning_domain.repeat(self.n_samples, 1).detach().cpu().numpy(),
            options={"eps": 1e-03},
        )
        if verbose:
            print(
                "ScipyBeamAlignment evaluated",
                self.n_samples,
                "(pathwise) posterior samples",
                res.nfev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyBeamAlignment evaluated",
                self.n_samples,
                "(pathwise) posterior sample jacobians",
                res.njev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyBeamAlignment took",
                res.nit,
                "steps in get_sample_optimal_tuning_configs().",
            )

        x_tuning_best_flat = torch.tensor(res.x)

        x_tuning_best = x_tuning_best_flat.reshape(
            self.n_samples, 1, -1
        )  # each row represents its respective sample's optimal tuning config


        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")

        xs = self.get_meas_scan_inputs(x_tuning_best, meas_scans, self.meas_dims)
        xs_exe = xs
        
        # evaluate posterior samples at input locations
        ys_exe_list = [sample_func(xs_exe).reshape(
            self.n_samples, 1+len(self.meas_dims), 1
        ) for sample_func in sample_funcs_list]
        ys_exe = torch.cat(ys_exe_list, dim=-1)
                            
        results_dict = {
            "xs_exe": xs_exe,
            "ys_exe": ys_exe,
            "x_tuning_best": x_tuning_best,
            "sample_funcs_list": sample_funcs_list,
        }

        return xs_exe, ys_exe, results_dict

    def evaluate_virtual_objective(self, model, x, bounds, n_samples=100, tkwargs:dict=None, use_mean=False):
        meas_scans = torch.index_select(
            bounds.T, dim=0, index=torch.tensor(self.meas_dims)
        )
        ndim = bounds.shape[1]
        tuning_dims = [i for i in range(ndim) if i not in self.meas_dims]
        x_tuning = torch.index_select(
            x, dim=-1, index=torch.tensor(tuning_dims)
        )
        if isinstance(model, ModelList):
            x_tuning = x_tuning.unsqueeze(1)
            xs = self.get_meas_scan_inputs(x_tuning, meas_scans, self.meas_dims)
            p = model.posterior(xs)
            ys = p.sample(torch.Size([n_samples])) # shape n_samples x n_points x (n_meas_scans + 1) x (1 or 2) (2 if x&y)
            rise = ys[...,1:,:] - ys[...,0:1,:] # shape n_samples x n_points x n_meas_scans x 1 or 2
            run = (meas_scans[:, 1] - meas_scans[:, 0]).reshape(-1,1).repeat(*rise.shape[:-2],1,rise.shape[-1]) # same shape as rise
            slope = rise/run
            misalignment = slope.pow(2).sum(dim=-1).sum(dim=-1,keepdim=True) # shape n_samples x n_points
        else:
            # TODO: add sample func list evaluation
            pass
        return misalignment

    def sample_funcs_misalignment(
        self,
        sample_funcs_list,
        x_tuning,  # n_samples x 1 x d tensor
        meas_dims,  # list of integers
        meas_scans,  # tensor of measurement device(s) scan inputs, shape: len(meas_dims) x 2
    ):
        """
        A function that computes the beam misalignment(s) through a set of measurement quadrupoles
        from a set of pathwise samples taken from a SingleTaskGP model of the beam centroid position with
        respect to some tuning devices and some measurement quadrupoles.

        arguments:
            sample_funcs_list: a list of pathwise posterior samples for x, y, or both 
                        from a SingleTaskGP model of the beam centroid positions (assumes Linear ProductKernel)
            x_tuning: a tensor of shape (n_samples x 1 x n_tuning_dims) where the nth entry defines a point in
                        tuning-parameter space at which to evaluate the misalignment of the nth
                        posterior pathwise sample given by post_paths
            meas_dims: the dimension indeces of our model that describe the quadrupole measurement devices
            meas_scans: a tensor of measurement scan inputs, shape len(meas_dims) x 2, where the nth row
                        contains two input scan values for the nth measurement quadrupole

         returns:
             misalignment: the sum of the squared slopes of the beam centroid model output with respect to the
                             measurement quads
             xs: the virtual scan inputs
             ys: the virtual scan outputs (beam centroid positions)

        NOTE: meas scans only needs to have 2 values for each device because it is expected that post_paths
                are produced from a SingleTaskGP with Linear ProductKernel (i.e. post_paths should have
                linear output for each dimension).
        """
        xs = self.get_meas_scan_inputs(x_tuning, meas_scans, meas_dims)
        sample_misalignments_sum_list = [] # list to store the sum of the samplewise misalignments in x, y or both
        sample_ys_list = [] # list to store the centroid positions for x, y or both
        for sample_func in sample_funcs_list:
            ys = sample_func(xs)
            ys = ys.reshape(self.n_samples, -1)

            rise = ys[:, 1:] - ys[:, 0].reshape(-1, 1)
            run = (meas_scans[:, 1] - meas_scans[:, 0]).T.repeat(ys.shape[0], 1)
            slope = rise / run

            misalignment = slope.pow(2).sum(dim=1)
            sample_misalignments_sum_list += [misalignment]
            sample_ys_list += [ys]
        
        total_misalignment = sum(sample_misalignments_sum_list)
        return total_misalignment, xs, sample_ys_list

    def get_meas_scan_inputs(self, x_tuning, meas_scans, meas_dims):
        # meas_scans = torch.index_select(
        #     bounds.T, dim=0, index=torch.tensor(self.meas_dims)
        # )    
        n_steps_meas_scan = 1 + len(meas_dims)
        n_tuning_configs = x_tuning.shape[0]

        # construct measurement scan inputs
        xs = torch.repeat_interleave(x_tuning, n_steps_meas_scan, dim=-2)

        for i in range(len(meas_dims)):
            meas_dim = meas_dims[i]
            meas_scan = meas_scans[i]
            full_scan_column = meas_scan[0].repeat(n_steps_meas_scan, 1)
            full_scan_column[i + 1, 0] = meas_scan[1]
            full_scan_column_repeated = full_scan_column.repeat(*x_tuning.shape[:-1], 1)

            xs = torch.cat(
                (xs[..., :meas_dim], full_scan_column_repeated, xs[..., meas_dim:]), dim=-1
            )

        return xs

    def sum_samplewise_misalignment_flat_x(
        self, sample_funcs_list, x_tuning_flat, meas_dims, meas_scans
    ):
        """
        A wrapper function that computes the sum of the samplewise misalignments for more convenient
        minimization with scipy.

        arguments:
            Same as post_path_misalignment() EXCEPT:

            x_tuning_flat: a FLATTENED tensor formerly of shape (n_samples x ndim) where the nth
                            row defines a point in tuning-parameter space at which to evaluate the
                            misalignment of the nth posterior pathwise samples given by sample_funcs_list

            NOTE: x_tuning_flat must be 1d (flattened) so the output of this function can be minimized
                    with scipy minimization routines (that expect a 1d vector of inputs)
            NOTE: samplewise is set to True to avoid unncessary computation during simultaneous minimization
                    of the pathwise misalignments.
        """

        x_tuning = x_tuning_flat.double().reshape(self.n_samples, 1, -1)

        return torch.sum(
            self.sample_funcs_misalignment(
                sample_funcs_list, x_tuning, meas_dims, meas_scans
            )[0]
        )