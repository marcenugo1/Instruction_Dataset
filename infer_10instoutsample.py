from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM,PeftModel
from datasets import load_dataset,load_from_disk
import torchinfo
import evaluate
import os
import torch
import pandas as pd
import json
import math
import random

no_vulnerabilities_messages = [
    "Upon examination, no vulnerabilities or security breaches were detected in the provided code.",
    "After thorough analysis, it seems that the code is free from any vulnerabilities or security risks.",
    "The code appears to be secure, with no evident vulnerabilities or potential security breaches.",
    "Based on the assessment, there are no indications of vulnerabilities or security breaches in the code.",
    "From the review conducted, it appears that the code is devoid of any vulnerabilities or security loopholes.",
    "No security vulnerabilities or breaches were found during the evaluation of the code.",
    "The code seems to be robust and secure, showing no signs of vulnerabilities or security risks.",
    "After careful scrutiny, it can be concluded that the code is devoid of any vulnerabilities or potential security threats.",
    "There are no apparent vulnerabilities or security breaches present in the analyzed code.",
    "The code seems to be well-protected against vulnerabilities and security breaches, with no notable issues detected."
]

def generate_response(model, tokenizer, query):
    # Ensure that the model and input tensors are on the same device
    device = next(model.parameters()).device
    
    # Tokenize inputs
    inputs = tokenizer(query, return_tensors="pt", padding=True, add_special_tokens=True)
    input_ids = inputs["input_ids"].to("cuda")
    
    # Generate output
    output = model.generate(input_ids=input_ids,max_new_tokens=200)
    
    # Decode and return response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def getPrompt(input,mode):
    
    inst = [
    "Evaluate the code for any potential security weaknesses and provide an explanation.",
    "Conduct a thorough examination of the code to identify and elucidate potential vulnerabilities.",
    "Perform a comprehensive analysis of the codebase, highlighting any possible security risks.",
    "Scrutinize the code for vulnerabilities and elucidate your findings.",
    "Assess the code's integrity and elucidate any potential points of vulnerability.",
    "Review the codebase and articulate any potential security flaws discovered.",
    "Delve into the code to uncover and explain potential vulnerabilities.",
    "Examine the code closely and elucidate any areas susceptible to security threats.",
    "Investigate the code for potential security loopholes and provide a detailed explanation.",
    "Scrutinize the code for potential exploits and elucidate your observations."
    ]


    if mode == "CWE":
        groundColumn = "label"
        # Format the code into decompiled_code_modified
        instruction = "Determine the CWE vulnerability code found in the code excerpt. If there is no CWE vulnerability answer 'non-vulnerable'. The code is the following:"
        text = f'''<s>[INST] {instruction}''' + f'''{input} [/INST]'''
        return text,groundColumn
    else:
        groundColumn = "comments"
        # Format the code into decompiled_code_modified
        #instruction = "Can you summarize this decompiled code? Also, if there is a vulnerability in this code, please describe the CWE:"
        instruction = random.choice(inst)
        text = f'''<s>[INST] {instruction}''' + f'''{input} [/INST]'''
        return text,groundColumn
        
def main():
  
    modelname = "/workspace/storage/LLM_testing/MVD_35K"
    
    model = AutoModelForCausalLM.from_pretrained(
            modelname,
            #device_map="auto",
            #load_in_8bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True
            )
    
    model.bfloat16().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    dataset = load_from_disk('/workspace/storage/finetune/MVD_alpaca_split')
    dataset_test = dataset['test']
    pred = []
    gt =[]
    iter = 0
    decomp_code = []
    # Iterate through each code
    for code in dataset_test:
        if iter >= 10:
            break 
        
        input = code["stripped_decompiled_code"]
    
        if not input:
            continue
        
        mode= "comments"
        split_token="[/INST]"
        
        prompt,ground = getPrompt(input,mode)
        
        # if code[ground]:
        #     ground = "CWE-" + str(int(code[ground]))
        # else:
        #     ground = 'non-vulnerable'
        
        
        if code["label"]:
            ground = code[ground]
        else:
            ground = random.choice(no_vulnerabilities_messages)
            
        if ground is None or ground == "[]":
            ground = ""
        gt.append(ground)
        decomp_code.append(input)
        response = generate_response(model, tokenizer, prompt)
        response_split = response.split(split_token)[-1].strip()
        pred.append(response_split)
        
        # Print the generated text
        print(f"Input:{input}")
        print(f"Ground truth output:\n{ground}")    
        print(f"Generated Answer:\n{response}")
        print(f"Generated Answer Split:\n{response_split}")
        print(f"sample:{iter}")
        iter+=1
    
    result = {
        "decompcode": decomp_code,
        "pred": pred,
        "gt": gt
    }
    
    resultFile = "result_MVD_10Samples.json"
    with open(resultFile, "w") as f:
        json.dump(result, f, indent=4)
        
    print("result has been saved in file")
        
    
if __name__ == "__main__":
    main()