# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: transform.py

import numpy as np
from abc import ABCMeta, abstractmethod
import cv2
import six

from tensorpack_utils import get_rng, log_once

__all__ = []


@six.add_metaclass(ABCMeta)
class Augmentor(object):
    """ Base class for an augmentor"""

    def __init__(self):
        self.reset_state()

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and not k.startswith('_'):
                    setattr(self, k, v)

    def reset_state(self):
        """ reset rng and other state """
        self.rng = get_rng(self)

    def augment(self, d):
        """
        Perform augmentation on the data.

        Args:
            d: input data

        Returns:
            augmented data
        """
        d, params = self._augment_return_params(d)
        return d

    def augment_return_params(self, d):
        """
        Augment the data and return the augmentation parameters.
        If the augmentation is non-deterministic (random),
        the returned parameters can be used to augment another data with the identical transformation.
        This can be used for, e.g. augmenting image, masks, keypoints altogether with the
        same transformation.

        Returns:
            (augmented data, augmentation params)
        """
        return self._augment_return_params(d)

    def _augment_return_params(self, d):
        """
        Augment the image and return both image and params
        """
        prms = self._get_augment_params(d)
        return (self._augment(d, prms), prms)

    def augment_with_params(self, d, param):
        """
        Augment the data with the given param.

        Args:
            d: input data
            param: augmentation params returned by :meth:`augment_return_params`

        Returns:
            augmented data
        """
        return self._augment(d, param)

    @abstractmethod
    def _augment(self, d, param):
        """
        Augment with the given param and return the new data.
        The augmentor is allowed to modify data in-place.
        """

    def _get_augment_params(self, d):
        """
        Get the augmentor parameters.
        """
        return None

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "imgaug.MyAugmentor(field1={self.field1}, field2={self.field2})"
        """
        try:
            argspec = inspect.getargspec(self.__init__)
            assert argspec.varargs is None, "The default __repr__ doesn't work for varargs!"
            assert argspec.keywords is None, "The default __repr__ doesn't work for kwargs!"
            fields = argspec.args[1:]
            index_field_has_default = len(fields) - (0 if argspec.defaults is None else len(argspec.defaults))

            classname = type(self).__name__
            argstr = []
            for idx, f in enumerate(fields):
                assert hasattr(self, f), \
                    "Attribute {} not found! Default __repr__ only works if attributes match the constructor.".format(f)
                attr = getattr(self, f)
                if idx >= index_field_has_default:
                    if attr is argspec.defaults[idx - index_field_has_default]:
                        continue
                argstr.append("{}={}".format(f, pprint.pformat(attr)))
            return "imgaug.{}({})".format(classname, ', '.join(argstr))
        except AssertionError as e:
            log_once(e.args[0], 'warn')
            return super(Augmentor, self).__repr__()

    __str__ = __repr__


class ImageAugmentor(Augmentor):
    """
    ImageAugmentor should take images of type uint8 in range [0, 255], or
    floating point images in range [0, 1] or [0, 255].
    """
    def augment_coords(self, coords, param):
        """
        Augment the coordinates given the param.

        By default, an augmentor keeps coordinates unchanged.
        If a subclass of :class:`ImageAugmentor` changes coordinates but couldn't implement this method,
        it should ``raise NotImplementedError()``.

        Args:
            coords: Nx2 floating point numpy array where each row is (x, y)
            param: augmentation params returned by :meth:`augment_return_params`

        Returns:
            new coords
        """
        return self._augment_coords(coords, param)

    def _augment_coords(self, coords, param):
        return coords


class TransformAugmentorBase(ImageAugmentor):
    """
    Base class of augmentors which use :class:`ImageTransform`
    for the actual implementation of the transformations.

    It assumes that :meth:`_get_augment_params` should
    return a :class:`ImageTransform` instance, and it will use
    this instance to augment both image and coordinates.
    """
    def _augment(self, img, t):
        return t.apply_image(img)

    def _augment_coords(self, coords, t):
        return t.apply_coords(coords)


@six.add_metaclass(ABCMeta)
class ImageTransform(object):
    """
    A deterministic image transformation, used to implement
    the (probably random) augmentors.

    This way the deterministic part
    (the actual transformation which may be common between augmentors)
    can be separated from the random part
    (the random policy which is different between augmentors).
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and not k.startswith('_'):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img):
        pass

    @abstractmethod
    def apply_coords(self, coords):
        pass


class ResizeTransform(ImageTransform):
    def __init__(self, h, w, newh, neww, interp):
        super(ResizeTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        assert img.shape[:2] == (self.h, self.w)
        ret = cv2.resize(
            img, (self.neww, self.newh),
            interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.neww * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.newh * 1.0 / self.h)
        return coords


class CropTransform(ImageTransform):
    def __init__(self, h0, w0, h, w):
        super(CropTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        return img[self.h0:self.h0 + self.h, self.w0:self.w0 + self.w]

    def apply_coords(self, coords):
        coords[:, 0] -= self.w0
        coords[:, 1] -= self.h0
        return coords


class WarpAffineTransform(ImageTransform):
    def __init__(self, mat, dsize, interp=cv2.INTER_LINEAR,
                 borderMode=cv2.BORDER_CONSTANT, borderValue=0):
        super(WarpAffineTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        ret = cv2.warpAffine(img, self.mat, self.dsize,
                             flags=self.interp,
                             borderMode=self.borderMode,
                             borderValue=self.borderValue)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        coords = np.concatenate((coords, np.ones((coords.shape[0], 1), dtype='f4')), axis=1)
        coords = np.dot(coords, self.mat.T)
        return coords


if __name__ == '__main__':
    shape = (100, 100)
    center = (10, 70)
    mat = cv2.getRotationMatrix2D(center, 20, 1)
    trans = WarpAffineTransform(mat, (130, 130))

    def draw_points(img, pts):
        for p in pts:
            try:
                img[int(p[1]), int(p[0])] = 0
            except IndexError:
                pass

    image = cv2.imread('cat.jpg')
    image = cv2.resize(image, shape)
    orig_image = image.copy()
    coords = np.random.randint(100, size=(20, 2))

    draw_points(orig_image, coords)
    print(coords)

    for k in range(1):
        coords = trans.apply_coords(coords)
        image = trans.apply_image(image)
    print(coords)
    draw_points(image, coords)

    # viz = cv2.resize(viz, (1200, 600))
    orig_image = cv2.resize(orig_image, (600, 600))
    image = cv2.resize(image, (600, 600))
    viz = np.concatenate((orig_image, image), axis=1)
    cv2.imshow("mat", viz)
    cv2.waitKey()
