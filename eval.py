from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model_custom = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map={'': 0},
)
model_original = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map={'': 1},
)
model_custom = PeftModel.from_pretrained(model_custom, "chinese-lora-alpaca", device_map={'': 0})
model_original = PeftModel.from_pretrained(model_original, "tloen/alpaca-lora-7b", device_map={'': 1})


def generate_instruction_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)


def evaluate(model_aaa, instruction, input=None):
    prompt = generate_instruction_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model_aaa.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Response:", output.split("### Response:")[1].strip())

# evaluate(model_custom, "中国的首都在哪里？")
# evaluate(model_original, "中国的首都在哪里？")

def get_answer(question):
    print(f"Q: {question}")
    print("Custom model:")
    evaluate(model_custom, question)
    print("Original model:")
    evaluate(model_original, question)

# question = "高考满分才750，怎么才能考上985？"
# get_answer(question)
# question = "\"被发下瀛州\"是什么意思？"
# get_answer(question)
# question = "历史上西施活了多少岁？"
# get_answer(question)
# question = "日本的首都在哪里？"
# get_answer(question)
question = "华中师范大学在哪里?"
get_answer(question)
