import pandas as pd
import random
import math

input_file = '/workspace/storage/experiment_datasets_finetune/MVD_35K.csv'
output_file = '/workspace/storage/experiment_datasets_finetune/MVD_35K_prompt.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)

instructions = [
    "Determine the CWE vulnerability code found in the provided code excerpt. If no CWE vulnerability is present, indicate 'non-vulnerable'. The code snippet is as follows:",
    "Identify any CWE vulnerability code present in the code snippet provided. If there are no vulnerabilities, specify 'non-vulnerable'. Here is the code excerpt:",
    "Please determine the CWE vulnerability code present in the provided code snippet. If no vulnerabilities are found, specify 'non-vulnerable'. The code snippet is given below:",
    "Locate any CWE vulnerability code present in the code excerpt provided. If no vulnerabilities are detected, specify 'non-vulnerable'. Here is the code snippet:",
    "Examine the code excerpt provided and determine any CWE vulnerability code. If there are none, indicate 'non-vulnerable'. The code excerpt is as follows:",
    "Review the code snippet provided and identify any CWE vulnerability code. If there are no vulnerabilities, specify 'non-vulnerable'. The code snippet is given below:",
    "Analyze the code excerpt provided and determine any CWE vulnerability code. If no vulnerabilities are found, indicate 'non-vulnerable'. The code excerpt is as follows:",
    "Inspect the provided code snippet and locate any CWE vulnerability code. If there are no vulnerabilities, specify 'non-vulnerable'. Here is the code excerpt:",
    "Evaluate the code snippet provided and identify any CWE vulnerability code. If no vulnerabilities are present, indicate 'non-vulnerable'. The code snippet is given below:",
    "Assess the provided code excerpt and determine any CWE vulnerability code. If there are none, specify 'non-vulnerable'. The code excerpt is as follows:",
    "Scrutinize the code excerpt provided and locate any CWE vulnerability code. If no vulnerabilities are found, specify 'non-vulnerable'. Here is the code excerpt:",
    "Examine the provided code snippet and identify any CWE vulnerability code. If there are no vulnerabilities, indicate 'non-vulnerable'. The code snippet is as follows:",
    "Investigate the code excerpt provided and determine any CWE vulnerability code. If no vulnerabilities are detected, specify 'non-vulnerable'. The code excerpt is given below:",
    "Analyze the provided code snippet and locate any CWE vulnerability code. If there are no vulnerabilities, specify 'non-vulnerable'. Here is the code snippet:",
    "Inspect the code excerpt provided and identify any CWE vulnerability code. If no vulnerabilities are present, indicate 'non-vulnerable'. The code excerpt is as follows:",
    "Review the provided code snippet and determine any CWE vulnerability code. If there are none, specify 'non-vulnerable'. The code snippet is given below:",
    "Examine the provided code excerpt and locate any CWE vulnerability code. If there are no vulnerabilities, specify 'non-vulnerable'. Here is the code excerpt:",
    "Assess the code snippet provided and identify any CWE vulnerability code. If there are no vulnerabilities, indicate 'non-vulnerable'. The code snippet is as follows:",
    "Scrutinize the provided code snippet and determine any CWE vulnerability code. If no vulnerabilities are found, specify 'non-vulnerable'. The code snippet is given below:",
    "Investigate the provided code excerpt and locate any CWE vulnerability code. If there are none, specify 'non-vulnerable'. Here is the code excerpt:"
]

explain = [
    "What potential security flaws do you notice in this code snippet?",
    "Can you identify any vulnerabilities in the provided code?",
    "Examine the code and point out any possible security risks.",
    "Are there any security loopholes in this code that you can detect?",
    "In your analysis, do you find any weaknesses in the code's security measures?",
    "Could you assess the code for any potential security weaknesses?",
    "Highlight any areas where the code might be susceptible to security breaches.",
    "Analyze the code and explain any potential vulnerabilities you find.",
    "Identify and discuss any potential exploits that could target this code.",
    "Describe any security risks associated with the given code segment.",
    "Explain how this code could be vulnerable to malicious attacks.",
    "Point out any security gaps that could be exploited in this code.",
    "Can you enumerate any potential threats to the security of this code?",
    "Discuss the code's susceptibility to security threats.",
    "Evaluate the code's safety and highlight any areas of concern.",
    "Analyze the code and elucidate any potential security vulnerabilities.",
    "Elaborate on the security risks inherent in this code snippet.",
    "Discuss potential security threats present in the code sample.",
    "Evaluate the code's resilience against potential security breaches.",
    "Are there any potential avenues for exploitation in this code?"
]

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

def get_random_from_array(options):
    return random.choice(options)

def add_prompt(row):
    code_field = 'stripped_decompiled_code'
    if row.name % 2 == 0:
        if not math.isnan(row['label']):
            label = "CWE-" + str(int(row['label']))
        else:
            label = 'non-vulnerable'
        return "<s>[INST] " + get_random_from_array(instructions) + " " + row[code_field].strip() + " [/INST]" + label + "</s>"
    else:
        if not math.isnan(row['label']):
            comments = row['comments']
        else:
            comments = get_random_from_array(no_vulnerabilities_messages)
        return "<s>[INST] " + get_random_from_array(explain) + " " + row[code_field].strip() + " [/INST]" + comments + "</s>"

# Add a new 'prompt' field using apply()
df['prompt'] = df.apply(add_prompt, axis=1)

# Write the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)
