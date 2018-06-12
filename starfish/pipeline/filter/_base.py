from starfish.image import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class FilterAlgorithmBase(AlgorithmBase):
    def filter(self, stack: ImageStack) -> None:
        """Performs in-place filtering on an ImageStack."""
        raise NotImplementedError()
