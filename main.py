import torch

from shabririgrape.dataloader import get_data, gen_loader
from shabririgrape.trainer import Trainer
from shabririgrape import MODE, DATASET
from utils.exp_manager import ExperimentManager, ExperimentCfg
from utils.plot import plot_confusion_matrix
from utils import ARTIFACTS


@ExperimentManager('config/exp.toml')
def main(cfg: ExperimentCfg) -> dict:
    """
    Main function.

    Args:
        cfg (Experiment): experiment configuration

    Returns:
        dict: dictionary of accuracy results
    """
    train_data_container = get_data(MODE.TRAIN)
    test_data_container = get_data(MODE.TEST)
    train_loader = gen_loader(train_data_container, batch_size=cfg.batch_size, mode=MODE.TRAIN,
                              use_aug=cfg.use_aug)
    test_loader = gen_loader(test_data_container, batch_size=cfg.batch_size, mode=MODE.TEST)

    trainer = Trainer(**cfg.trainer)
    print(trainer.model)

    trainer.train(train_loader, test_loader, epochs=cfg.epochs)

    experiment_name = f'{cfg.name}_{cfg.exp_value}'

    confusion_matrix = trainer.compute_confusion_matrix(
        trainer.collector.test_output, trainer.collector.test_target
    )
    plot_confusion_matrix(
        f"{ARTIFACTS.RESULT}/{ARTIFACTS.CONFUSION_MATRIX}_{experiment_name}{ARTIFACTS.IMG_EXT}",
        experiment_name,
        confusion_matrix,
        DATASET.NUM_CLASSES
    )

    torch.save(trainer.model.state_dict(), f"{ARTIFACTS.MODEL}/{experiment_name}{ARTIFACTS.MODEL_EXT}")

    return {
        ARTIFACTS.TRAIN: trainer.collector.train_acc,
        ARTIFACTS.TEST: trainer.collector.test_acc
    }


if __name__ == '__main__':
    main()
