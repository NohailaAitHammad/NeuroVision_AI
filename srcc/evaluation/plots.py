from pathlib import Path
import matplotlib.pyplot as plt

def plot_training_history(history, output_dir):
      output_dir = Path(output_dir)
      output_dir.mkdir(parents=True, exist_ok=True)

      history_data = history.history

      #Courbe d'accuracy

      plt.figure(figsize=(8,5))
      plt.plot(history_data['accuracy'], label ='Train accuracy')

      if 'val_accuracy' in history_data:
            plt.plot(history_data['val_accuracy'], label ='Validation accuracy')

      plt.title('Accuracy - Fine Tuning MobileNetV2')
      plt.xlabel('Epoque')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(output_dir / "accuracy.png", dpi=150)
      plt.show()
      plt.close()


      #Courbe de loss
      plt.figure(figsize=(8,5))
      plt.plot(history_data['loss'], label ='Train loss')
      
      if 'val_loss' in history_data:
            plt.plot(history_data['val_loss'], label ='Validation loss')

      plt.title('Loss - Fine Tuning MobileNetV2')
      plt.xlabel('Epoque')
      plt.ylabel('Loss')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(output_dir / "loss.png", dpi=150)
      plt.show()
      plt.close()
