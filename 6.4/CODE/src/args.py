import argparse
import logging

logger = logging.getLogger(__name__)

def set_and_get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--ref_file_path',
                        help='Chemin vers le fichier excel des referentiels',
                        default='.'
                        )
    parser.add_argument('--use_json',
                        action='store_true',
                        help='Utiliser JSON pour les références à la place d\'Excel',
                        default=False)

    parser.add_argument('-m', '--mode',
                        help="Mode d'exécution",
                        default='.'
                        )

    args = parser.parse_args()

    return args