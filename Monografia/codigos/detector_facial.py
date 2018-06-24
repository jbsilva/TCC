#!/usr/bin/env python3
# -*-*- encoding: utf-8 -*-*-
# Created: Sat, 26 May 2018 19:32:23 -0300

"""
Detector de faces usando a biblioteca OpenCV.

Uso: detector_facial.py [-c CLASSIFICADOR]

Argumentos opcionais:
  -c CLASSIFICADOR, --classificador CLASSIFICADOR
                        Caminho para classificador em cascata
"""

import sys
import argparse
import cv2


__author__ = "Julio Batista Silva"
__copyright__ = "Copyright (c) 2018, Julio Batista Silva"
__license__ = "GPL v3"
__version__ = "1.0"
__email__ = "julio@juliobs.com"


DEFAULT_CLASSIFIER = 'haarcascades/haarcascade_frontalface_default.xml'


def main(ARGS):
    # Carrega o classificador em cascata passado por parâmetro
    classificador = cv2.CascadeClassifier(ARGS.classificador)

    # Captura vídeo da câmera
    camera = cv2.VideoCapture(0)

    # Detecta faces até a tecla ESC ser pressionada
    while camera.isOpened():
        _, imagem = camera.read()

        # Cria cópia em tons de cinza da imagem
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        # Detecta as faces usando a imagem em tons de cinza
        faces = classificador.detectMultiScale(
            cinza, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        # Desenha um retângulo em volta da face detectada
        for (x, y, w, h) in faces:
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Mostra imagem na tela
        cv2.imshow('Detector de Faces', imagem)

        # Libera a câmera e fecha a janela ao pressionar a tecla ESC
        if cv2.waitKey(1) == 27:
            camera.release()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detecta faces na câmera.')
    parser.add_argument('-c', '--classificador', type=str,
                        default=DEFAULT_CLASSIFIER,
                        help='Caminho para classificador em cascata')
    parser.add_argument('--versao', action='version',
                        version='%(prog)s v' + __version__)
    ARGS = parser.parse_args()

    sys.exit(main(ARGS))
