#!/usr/bin/env bash
mkdir data
cd data
wget http://www.statmt.org/europarl/v7/fr-en.tgz
tar zxvf fr-en.tgz
rm fr-en.tgz
