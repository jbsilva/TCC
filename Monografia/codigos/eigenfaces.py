#!/usr/bin/env python3
# -*-*- encoding: utf-8 -*-*-
# Created: Sun, 17 Jun 2018 16:19:23 -0300

"""
Reconhece faces usando Eigenfaces

uso: eigenfaces.py [-c CLASSIFICADOR]
                     [-t DIR_TREINO]
                     [-i [IMAGENS [IMAGENS ...]]]
                     [-o DIR_SAIDA]
"""


import os
import sys
import argparse
import logging
import numpy as np
import cv2


__author__ = "Julio Batista Silva"
__copyright__ = "Copyright (c) 2018, Julio Batista Silva"
__license__ = "GPL v3"
__version__ = "1.0"
__email__ = "julio@juliobs.com"


SCRIPT_DIR = os.path.dirname(__file__)
SCRIPT_NAME = os.path.basename(__file__)
DEFAULT_CLASSIFIER = 'haarcascades/haarcascade_frontalface_default.xml'
DEFAULT_IMGS_PATH = '/Users/julio/Pictures/Face_databases/ATT'
VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)


def le_imagens_de_treino(path, tam=None):
    '''Lê imagens do diretório path, redimensiona para tam e retorna uma tupla
    com três listas: imagens, ids e nomes'''

    imagens, ids, nomes = [], [], []
    qtd = 0

    pessoas = [x for x in os.listdir(path)
               if os.path.isdir(os.path.join(path, x))]

    for pessoa in pessoas:
        for foto in [os.path.join(path, pessoa, x) for x in
                     os.listdir(os.path.join(path, pessoa))]:
            img = cv2.imread(foto, cv2.IMREAD_GRAYSCALE)

            # Redimensiona imagem
            if tam is not None:
                img = cv2.resize(img, tam)

            imagens.append(np.asarray(img, dtype=np.uint8))
            ids.append(qtd)
        nomes.append(pessoa)
        qtd += 1

    return (imagens, ids, nomes)


def detecta_faces(imagem, classificador):
    # Carrega o classificador em cascata passado por parâmetro
    classif = cv2.CascadeClassifier(classificador)

    # Detecta as faces na imagem
    faces = classif.detectMultiScale(
        imagem, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

    return faces


def main(ARGS):

    # Obtém imagens de treino com ids e nomes
    X, y, nomes = le_imagens_de_treino(ARGS.dir_treino, (200, 200))

    # Treina o reconhecedor de faces
    reconhecedor = cv2.face.EigenFaceRecognizer_create()
    reconhecedor.train(np.asarray(X), np.asarray(y))

    # Para estatística
    acertos = 0
    erros = 0
    falhas = 0

    for img_path in ARGS.imagens:
        imagem = cv2.imread(img_path)

        if imagem is None:
            logging.debug(f"Não conseguiu abrir {img_path}.")
            falhas += 1
            continue

        # Cria cópia em tons de cinza da imagem
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        faces = detecta_faces(cinza, ARGS.classificador)
        logging.debug(f"{len(faces)} faces detectadas.")

        # Gabarito
        nome_certo = os.path.split(os.path.split(img_path)[0])[1]

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Corta e redimensiona face
                face = cinza[x:x + w, y:y + h]
                face = cv2.resize(
                    face, (200, 200), interpolation=cv2.INTER_LINEAR)

                # Tenta reconhecer a face. Se conseguir, escreve nome na imagem
                try:
                    # Confiança aceitável: abaixo de 5000
                    num, confianca = reconhecedor.predict(face)
                    nome = nomes[num]
                    logging.debug(f"Reconhecido: {nome}, "
                                  f"Confiança: {confianca:.2f}, "
                                  f"Correto: {nome_certo}")

                    # Desenha um retângulo em volta da face detectada
                    cv2.rectangle(
                        imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Escreve nome da pessoa reconhecida.
                    # Verde: correto, Vermelho: errado
                    if nome == nome_certo:
                        acertos += 1
                        cv2.putText(imagem, nome, (x, y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, VERDE, 2)
                    else:
                        cv2.putText(imagem, nome, (x, y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, VERMELHO, 2)
                        erros += 1

                except BaseException:
                    logging.exception("Não reconheceu")
                    falhas += 1

                # Grava imagem com o resultado da detecção
                pasta = ARGS.dir_saida if ARGS.dir_saida else os.path.dirname(
                    img_path)
                nome = f"{nome_certo}-{nome}-"
                    f"{os.path.splitext(os.path.basename(img_path))[0]}.jpg"
                cv2.imwrite(os.path.join(pasta, nome), imagem)
                logging.debug(f"Gravou em {os.path.join(pasta, nome)}.")

    logging.info(f"Acertos: {acertos}, Erros: {erros}, Falhas: {falhas}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Reconhece faces usando Eigenfaces')
    parser.add_argument('-c', '--classificador',
                        default=DEFAULT_CLASSIFIER,
                        help='Caminho para classificador em cascata')
    parser.add_argument('-t', '--dir_treino',
                        default=DEFAULT_IMGS_PATH,
                        help='Caminho da pasta com imagens de treino')
    parser.add_argument('-i', '--imagens', nargs='*',
                        help='Caminho para imagens a serem reconhecidas')
    parser.add_argument(
        '-o',
        '--dir_saida',
        help='Caminho onde imagens com reconhecimento serão salvas')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    ARGS = parser.parse_args()

    logging.basicConfig(
        filename=os.path.join(SCRIPT_DIR, SCRIPT_NAME + '.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')

    logging.debug("===================Iniciou o script===================")

    try:
        sys.exit(main(ARGS))
    except KeyboardInterrupt:
        logging.debug("=====================Interrompido=====================")
        sys.exit(0)
    except SystemExit:
        logging.debug("==================Finalizou o script==================")
    except BaseException:
        logging.exception('Ocorreu um erro')
