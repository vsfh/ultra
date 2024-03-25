# Ultralytics YOLO 🚀, AGPL-3.0 license

# from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, OBBModel, PoseModel, SegmentationModel
from ultralytics.models import YOLO
# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
import sys
from pathlib import Path
from typing import Union

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.hub.utils import HUB_WEB_ROOT
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import ASSETS, LOGGER, RANK, callbacks, checks, emojis, yaml_load
# from ultralytics.engine.exporter import Exporter
from exporter import Exporter
import sys
sys.path.append('.')
from cls.train import ClassificationTrainer
from cls.predict import ClassificationPredictor
from cls.val import ClassificationValidator

from detect.train import DetectionTrainer, PoseDetectionModel
from detect.predict import DetectionPredictor

class YOLO(yolo.model.Model):
    """YOLO (You Only Look Once) object detection model."""
    def predict(
        self,
        source = None,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ) -> list:
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode. It also provides support for SAM-type models
        through 'prompts'.

        The method sets up a new predictor if not already present and updates its arguments with each call.
        It also issues a warning and uses default assets if the 'source' is not provided. The method determines if it
        is being called from the command line interface and adjusts its behavior accordingly, including setting defaults
        for confidence threshold and saving behavior.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): The source of the image for making predictions.
                Accepts various types, including file paths, URLs, PIL images, and numpy arrays. Defaults to ASSETS.
            stream (bool, optional): Treats the input source as a continuous stream for predictions. Defaults to False.
            predictor (BasePredictor, optional): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor. Defaults to None.
            **kwargs (any): Additional keyword arguments for configuring the prediction process. These arguments allow
                for further customization of the prediction behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor is not properly set up.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (sys.argv[0].endswith("yolo") or sys.argv[0].endswith("ultralytics")) and any(
            x in sys.argv for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': ClassificationModel,
                'trainer': ClassificationTrainer,
                'validator': ClassificationValidator,
                'predictor': ClassificationPredictor, },
            'detect': {
                'model': PoseDetectionModel,
                'trainer': DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': DetectionPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, },
            'obb': {
                'model': OBBModel,
                'trainer': yolo.obb.OBBTrainer,
                'validator': yolo.obb.OBBValidator,
                'predictor': yolo.obb.OBBPredictor, }, }
