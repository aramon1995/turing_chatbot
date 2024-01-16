from torch import cuda, LongTensor, FloatTensor, eq
from transformers import StoppingCriteria, StoppingCriteriaList

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'



# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    
    def __init__(self, stop_token_ids) -> None:
        super().__init__()
        self.stop_token_ids = stop_token_ids
        
    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False