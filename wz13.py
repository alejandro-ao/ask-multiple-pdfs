from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class wizardVicuna13(LLM):
   
    @property
    def _llm_type(self) -> str:
        return "custom"
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
   
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
  
        model_name_or_path = "TheBloke_wizard-vicuna-13B-SuperHOT-8K-GPTQ"
        model_basename = "wizard-vicuna-13b-superhot-8k-GPTQ-4bit-128g.no-act.order"

        use_triton = False

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                    device_map={'': 0},
                use_triton=use_triton,
                quantize_config=None)

        model.seqlen = 8192
        prompt_template=f'''Question: {prompt}'''

        # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        logging.set_verbosity(logging.CRITICAL)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        text= pipe(prompt_template)[0]['generated_text'][len(prompt) :]
        print(text) 

        if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
        return text
