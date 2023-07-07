#pip install einops 
#pip install git+https://github.com/PanQiWei/AutoGPTQ@v0.2.1

from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

# model_name_or_path = "TheBloke/wizard-vicuna-13B-SuperHOT-8K-GPTQ"
model_name_or_path="TheBloke_wizard-vicuna-13B-SuperHOT-8K-GPTQ"
model_basename = "wizard-vicuna-13b-superhot-8k-GPTQ-4bit-128g.no-act.order"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device_map='auto',
        use_triton=use_triton,
        quantize_config=None)

model.seqlen = 8192

# Note: check the prompt template is correct for this model.
prompt = "Tell me about AI"
prompt_template=f'''USER: {prompt}
ASSISTANT:'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])