from utils import device, StopOnTokens
import transformers
from torch import bfloat16, LongTensor
from transformers import StoppingCriteriaList
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


class TinyLLama:
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    stop_list = ["<|user|>", "\n```\n"]
    question_generator_template = PromptTemplate.from_template(
        "<|system|>\n"
        "You are an assistant who takes a chat history and a flow up question and "
        "combine the chat history and follow up question into "
        "a standalone question.</s>"
        "\n<|user|>\n"
        "Chat History: {chat_history}\n"
        "Follow up question: {question}</s>"
        "\n<|assistant|>\n"
    )

    qa_template = PromptTemplate.from_template(
        "<|system|>\n"
        "You are an assistant who reads a context and respond the user question based on it, "
        "if the answer is not in the context do not make it up"
        "\nContext:\n"
        "{context}\n"
        "{chat_history}</s>"
        "\n<|user|>\n"
        "{question}</s>"
        "\n<|assistant|>\n"
    )

    def __init__(self):
        if device != "cpu":
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16,
            )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=bnb_config if device != "cpu" else None,
        )

        model.to(device)

        # enable evaluation mode to allow model inference
        model.eval()
        print(f"Model loaded on {device}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)

        stop_token_ids = [
            LongTensor(tokenizer(x)["input_ids"]).to(device)
            for x in self.stop_list
        ]
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task="text-generation",
            stopping_criteria=stopping_criteria,
            temperature=0.05,
            max_new_tokens=512,
            repetition_penalty=1.1,
        )
        self.llm = HuggingFacePipeline(pipeline=generate_text)
