from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str, labels: list[str]):
        """
        Detect objects in an image.

        Args:
            image_path: Path to the input image.
            labels: List of labels to detect.

        Returns:
            List of detections.
        """
        pass