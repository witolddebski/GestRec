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

* Distributor -
* VendingMachine -
* Display -

The app implements an MVC design pattern with these classes.