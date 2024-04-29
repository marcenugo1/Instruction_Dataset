import evaluate
from bert_score import score as bert_score
import json 

def main():
    
    # Load JSON data from file
    with open('/workspace/storage/result_MVD_35K_describe2.json', 'r') as f:
        data = json.load(f)
        
    # Extract 'decompcode' and 'red' lists
    all_predictions = data.get('pred', [])
    all_references = data.get('gt', [])
    # Replace None with empty string
    #references = [ref if ref is not None else '' for ref in references]
    
    predictions = []
    references = []


    for i in range(len(all_references)):
        if all_references[i] and all_predictions[i]:  # Check if the value in gt is not empty
            predictions.append(all_predictions[i])
            references.append(all_references[i])
            
    # for i in range(len(all_references)):
    #     if all_references[i] != "" and all_references[i] != "Based on the assessment, there are no indications of vulnerabilities or security breaches in the code.":  # Check if the value in gt is not empty
    #         predictions.append(all_predictions[i])
    #         references.append(all_references[i])
        
    print(len(references))
    # Load evaluation metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    bert_metric = evaluate.load("bertscore")

    # Compute BERTScore
    bert_score_precision, bert_score_recall, bert_score_f1 = bert_score(predictions, references, lang="en", verbose=False)

    # Compute BLEU
    bleu_score = bleu_metric.compute(predictions=predictions, references=references,smooth=True)['bleu']
    
    # Compute Rouge-L
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)['rougeL']
    
    
    print(f'Bleu score: {bleu_score}\n' +
            f'Rouge: {rouge_score}\n' +
            f'Bert_score_precision: {bert_score_precision.mean().item()}\n' +
            f'Bert_score_recall: {bert_score_recall.mean().item()}\n' +
            f'Bert_score_f1: {bert_score_f1.mean().item()}\n')
    
if __name__ == "__main__":
    main()