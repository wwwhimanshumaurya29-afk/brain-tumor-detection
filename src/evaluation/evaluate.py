
from src.config import (
    DEVICE,
    MODEL_DIR,
    RESULTS_DIR,
    NUM_CLASSES,
    PRETRAINED,
    CLASS_NAMES
)

from src.data.data_loader import create_data_loaders
from src.models.model import create_model

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load V2 checkpoint
    checkpoint_path = MODEL_DIR / "best_model_v2.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    # Load test data only
    _, _, test_loader = create_data_loaders()

    # Create model
    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED
    ).to(DEVICE)

    # Load best V2 model
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded V2 checkpoint")
    print("Best epoch:", checkpoint["epoch"])
    print("Validation loss:", checkpoint["validation_loss"])
    print("Validation accuracy:", checkpoint["validation_accuracy"])

    # Evaluation
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    precision = precision_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )

    recall = recall_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )

    f1 = f1_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )

    matrix = confusion_matrix(
        all_labels, all_predictions,
        labels=[0, 1, 2]
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    # Display results
    print("\nBRAIN TUMOR MODEL V2 EVALUATION")
    print("==============================")

    print(f"Test Samples: {len(all_labels)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nCONFUSION MATRIX")
    print(matrix)

    print("\nCLASSIFICATION REPORT")
    print(report)

    # Save results separately from V1
    result_path = RESULTS_DIR / "evaluation_results_v2.txt"

    with open(result_path, "w", encoding="utf-8") as file:

        file.write("BRAIN TUMOR MODEL V2 EVALUATION\n")
        file.write("==============================\n")

        file.write(f"Checkpoint Epoch: {checkpoint['epoch']}\n")
        file.write(
            f"Validation Loss: {checkpoint['validation_loss']:.4f}\n"
        )
        file.write(
            f"Validation Accuracy: {checkpoint['validation_accuracy']:.4f}\n"
        )

        file.write(f"Test Samples: {len(all_labels)}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1-score: {f1:.4f}\n")

        file.write("\nCONFUSION MATRIX\n")
        file.write(str(matrix))

        file.write("\n\nCLASSIFICATION REPORT\n")
        file.write(report)

    print("\nResults saved:", result_path)


if __name__ == "__main__":
    evaluate_model()
