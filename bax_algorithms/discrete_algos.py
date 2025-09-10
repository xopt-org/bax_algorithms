from .base_discrete import DiscreteSubsetAlgorithm
from .algo_fns import global_opt, single_level_band, multi_level_band
from pydantic import Field

class GlobalOpt(DiscreteSubsetAlgorithm):
    minimize: bool = Field(True, 
                           description="If true, minimize function, otherwise maximize")
    
    def __init__(self, **data):
        data["algo_fn"] = global_opt
        data["algo_kwargs"] = {"minimize": data["minimize"]}
        super().__init__(**data)
        
class SingleLevelBand(DiscreteSubsetAlgorithm):
    min_val: float = Field(..., 
                           description="Min value of band")
    max_val: float = Field(..., 
                           description="Max value of band")
    
    def __init__(self, **data):
        data["algo_fn"] = single_level_band
        data["algo_kwargs"] = {"min_val": data["min_val"], "max_val": data["max_val"]}
        super().__init__(**data)
        
class MultiLevelBand(DiscreteSubsetAlgorithm):
    bounds_list: list = Field(..., 
                              description="List of bounds for multi-level band")
    
    def __init__(self, **data):
        data["algo_fn"] = multi_level_band
        data["algo_kwargs"] = {"bounds_list": data["bounds_list"]}
        super().__init__(**data)