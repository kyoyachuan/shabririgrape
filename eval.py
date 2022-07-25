import torch

from shabririgrape.dataloader import get_data, gen_loader
from shabririgrape.trainer import Trainer
from shabririgrape import MODE
from utils.exp_manager import ExperimentManager, ExperimentCfg
from utils import ARTIFACTS


@ExperimentManager('config/exp.toml', plot_within_experiment_group=False)
def eval(cfg: ExperimentCfg) -> dict:
    """
    Evaluation function.

    Args:
        cfg (Experiment): experiment configuration

    Returns:
        dict: dictionary of accuracy results
    """
    experiment_name = f'{cfg.name}_{cfg.exp_value}'

    test_data_container = get_data(MODE.TEST)
    test_loader = gen_loader(test_data_container, batch_size=cfg.batch_size, mode=MODE.TEST)

    trainer = Trainer(**cfg.trainer)
    trainer.model.load_state_dict(torch.load(f"{ARTIFACTS.MODEL}/{experiment_name}{ARTIFACTS.MODEL_EXT}"))
    print(trainer.model)

    trainer.validate_epoch(test_loader)

    return {
        ARTIFACTS.TEST: trainer.collector.test_acc
    }


if __name__ == '__main__':
    eval()
