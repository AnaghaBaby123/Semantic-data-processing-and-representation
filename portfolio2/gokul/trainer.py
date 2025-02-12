import torch
import torch.nn.functional as F

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.notebook import tqdm

I2L = {0: "NotOnion", 1: "Onion"}


def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(data_loader, unit='batch')
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        probabilities = F.softmax(outputs.logits, dim=-1)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        preds = torch.argmax(probabilities, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()

        pbar.set_description(desc=f"batch_loss={loss:.4f}")

        # break
    
    return total_loss / len(data_loader), (100. * total_correct / total_samples)


def evaluate_model(model, data_loader, device, return_reports=False, return_misclassified=False):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    misclassified_examples = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['text'] if 'text' in batch else None
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            probabilities = F.softmax(outputs.logits, dim=1)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(probabilities, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if return_misclassified and texts is not None:
                incorrect_mask = preds != labels
                if incorrect_mask.any():
                    for idx in range(len(preds)):
                        if incorrect_mask[idx]:
                            misclassified_examples.append({
                                'text': texts[idx],
                                'true_label': I2L[labels[idx].item()],
                                'predicted_label': I2L[preds[idx].item()],
                                'probabilities': probabilities[idx].cpu().numpy()
                            })
            
            all_preds.extend([I2L[pred] for pred in preds.cpu().numpy()])
            all_labels.extend([I2L[label] for label in labels.cpu().numpy()])
            
            # break

    if return_reports:
        report = classification_report(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
    else:
        report = None
        cm = None
    
    results = {
        'loss': total_loss / len(data_loader),
        'accuracy': 100. * total_correct / total_samples,
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    if return_misclassified:
        results['misclassified_examples'] = misclassified_examples
    
    return results


def train_model(model, tokenizer, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train one epoch
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        
        # Record losses
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on test set
        test_results = evaluate_model(model, val_loader, device, return_reports=False, return_misclassified=False)
        test_losses.append(test_results['loss'])
        test_accuracies.append(test_results['accuracy'])

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}')
        print(f"Test Loss: {test_results['loss']:.4f}, Test Accuracy: {test_results['accuracy']:.2f}")

        if test_results['accuracy'] > best_accuracy:
            best_accuracy = test_results['accuracy']
            output_dir = Path("model/best")
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"New best model saved! Validation Accuracy: {test_results['accuracy']:.2f}%")
        
        if epoch == num_epochs-1:
            output_dir = Path("model/final")
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Final model saved! Validation Accuracy: {test_results['accuracy']:.2f}%")

        print("-" * 60)
        # break

    return train_losses, train_accuracies, test_losses, test_accuracies


def freeze_transformer_layers(model):
    """Freeze all layers except the classification head"""
    
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Only classifier parameters will be updated during training
    for param in model.classifier.parameters():
        param.requires_grad = True


def inference(model, tokenizer, text, device, max_length=512):

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text
    
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        probabilities = F.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        probs_np = probabilities.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        
        results = []
        for idx, (text, pred, prob) in enumerate(zip(texts, preds_np, probs_np)):
            result = {
                'text': text,
                'prediction': I2L[pred],
                'probabilities': {I2L[i]: float(p) for i, p in enumerate(prob)},
                'confidence': float(prob[pred])
            }
            results.append(result)
            
    if isinstance(text, str):
        return results[0]
    
    return results