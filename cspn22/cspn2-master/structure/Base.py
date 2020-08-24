"""
Created on November 24, 2018

@author: Alejandro Molina
"""
from spn.structure.Base import Node


class Or(Node):
    def __init__(self, children=None):
        Node.__init__(self)

        if children is None:
            children = []
        self.children = children
