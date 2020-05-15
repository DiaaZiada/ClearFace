#!/bin/bash
cd gender && cmake -H. -Bbuild && cd build && make && cd .. && cd ..
cd expression && cmake -H. -Bbuild && cd build && make && cd .. && cd ..
cd multiple && cmake -H. -Bbuild && cd build && make && cd .. && cd ..

