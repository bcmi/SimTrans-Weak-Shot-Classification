from data.bird_helper import CUBHelper
from data.car_helper import CarHelper
from data.Air_helper import AirHelper
from data.webvision_helper import WebVisionHelper


def get_data_helper(args):
    if 'CUB' in args.data_path:
        return CUBHelper(args)
    elif 'Car' in args.data_path:
        return CarHelper(args)
    elif 'Air' in args.data_path:
        return AirHelper(args)
    elif 'WebVision' in args.data_path:
        return WebVisionHelper(args)
    else:
        raise NotImplementedError
