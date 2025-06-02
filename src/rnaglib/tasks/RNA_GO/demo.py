from torch.utils.tensorboard import SummaryWriter
from rnaglib.dataset_transforms import RandomSplitter
from rnaglib.tasks import RNAGo
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

# Initialisation TensorBoard
writer = SummaryWriter('runs/exp1')

ta = RNAGo(
    root="RNA_GO_random",
    splitter=RandomSplitter(),
    recompute=False,
    debug=False
)

# Adding representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
print("Splitting Dataset")
ta.get_split_loaders(batch_size=1)

info = ta.describe()

model = PygModel(
    ta.metadata["num_node_features"],
    num_classes=len(ta.metadata["label_mapping"]),
    graph_level=True,
    multi_label=True
)
model.configure_training(learning_rate=0.0001)

epochs = 20
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    # Tu peux adapter ça selon comment ton modèle gère les epochs
    # Par exemple, entraîner une epoch complète ici, récupérer la loss
    model.train_model(ta, epochs=1)  # entraînement 1 epoch à la fois
    
    # Supposons que model.train_model met à jour un attribut `last_loss`
    loss = getattr(model, "last_loss", None)  # adapte selon ta classe
    if loss is not None:
        writer.add_scalar('Loss/train', loss, epoch)
    
    # Évaluer les métriques à chaque epoch (optionnel)
    metrics = model.evaluate(ta)
    for metric_name, value in metrics.items():
        writer.add_scalar(f'Test/{metric_name}', value, epoch)

writer.close()
