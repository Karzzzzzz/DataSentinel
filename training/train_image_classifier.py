from modules.image_ai.image_training import train_custom_image_classifier

if __name__ == "__main__":
    result = train_custom_image_classifier()
    print(f"Held-out accuracy: {result['accuracy']:.3f}")
    print(f"Training samples: {result['training_samples']}")
    print(f"Test samples: {result['test_samples']}")
    print(f"Categories: {', '.join(result['classes'])}")
    print(f"Saved model: {result['model_path']}")
    print(result["note"])
