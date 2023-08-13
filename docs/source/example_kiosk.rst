Example Application
===================

Part of this library is an exemplary application showcasing its capabilities called Kiosk App.

Kiosk App simulates a vending machine program with simple logic for choosing and ordering a drink
from currently available offer. You can run it by importing the file and running ``launch()`` method
of a ``VendingMachine`` object:

.. code-block:: python

    >>> import gesture.kiosk_app as kiosk
    >>> machine1 = kiosk.VendingMachine()
    >>> machine1.launch()

The module contains three main classes:

* ``Distributor`` - simulates business logic (current offering, managing stock)
* ``VendingMachine`` - receives user's gestures and calls appropriate methods of ``Distributor``
* ``Display`` - displays feedback & instructions to the user

The application is using OpenCV to get images from camera to be analyzed by the library.

``VendingMachine`` is an example of decoupling business logic from gesture interpretation and chaining.
For example, in order to purchase a beverage, the user needs to choose a beverage with one gesture
and then confirm selection with another one. This chain of gestures is interpreted by ``VendingMachine``,
which then calls ``purchase()`` on ``Distributor``.

Inside ``VendingMachine`` you will also find an example of mapping class IDs onto a custom set of labels
in form of ``self.gesture_dict``.