Getting Started
=====

Installation
------------

Begin by cloning the repository from `github <https://github.com/witolddebski/gestRec/tree/master>`_.

Classifying Gestures
--------------------

To use GestRec, import the package and create the ``Recognizer`` object:

.. code-block:: python

   import gesture
   rec = Recognizer()

.. note::
    changing the underlying model requires re-instantiating the object.

The default model loaded is mobilenet v3 large []. Next,all you need is an image
to be processed. A test image is supplied with this package:

.. code-block:: python

    from PIL import Image
    image = Image.open("test_images/16.jpg")

Then, use the ``__call__`` method to perform inference:

.. code-block:: python

    print(rec(image))
