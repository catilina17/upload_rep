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
    parser.add_argument('-bse', '--batch_size_ech',
                        help="Mode d'exécution",
                        default=5000
                        )

    parser.add_argument('-bsm', '--batch_size_nmd',
                        help="Mode d'exécution",
                        default=10000
                        )

    args = parser.parse_args()

    return args