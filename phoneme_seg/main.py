from train_test import train_test, test
from model import Classifier
from utils import load_model, load_from_checkpoint
from dataloader import get_dloaders
import hydra, os, logging, random, numpy as np, torch


@hydra.main(config_path='../conf', config_name='phoneme_seg')
def main(cfg):

    logger = logging.getLogger(__name__)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.autograd.set_detect_anomaly(True)

    device = torch.device(cfg.device)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device.split(':')[1]

    logger.info("Model base checkpoint is {}".format(cfg.base_ckpt_path))

    logger.info("Instantiating model...")
    model, layers = load_model(cfg, device)
    model.cfg.fp16 = cfg.fp16
    classifier = Classifier(mode=cfg.mode, n_layers=12)

    model = model.to(device)
    classifier = classifier.to(device)

    trainloader, valloader, testloader = get_dloaders(
        cfg=cfg, layers=layers, logger=logger, g=g)
    train_test(cfg, model, classifier, trainloader, valloader, testloader, logger)
    # model, _, classifier, _, metrics = load_from_checkpoint(cfg, device, cfg.test_ckpt_path)
    # test(model, classifier, testloader, cfg, logger, device)

if __name__ == "__main__":
    main()