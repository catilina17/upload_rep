import json
import os
import sys

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def read_json_file(json_file_path):
    """ Lire un fichier JSON et retourner les données. """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_value_from_named_ranged(json_data, key_path):
    """
    Vérifie la présence d'une clé spécifique dans le dictionnaire JSON et retourne sa valeur,
    même si cette valeur est vide (comme une chaîne vide, une liste vide, etc.).

    Args:
        json_data (dict): Le dictionnaire JSON à vérifier.
        key_path (list): Chemin de la clé à vérifier dans le JSON (liste de clés).
        alert_message (str): Message d'alerte si la clé est absente ou la valeur est None.

    Returns:
        La valeur de la clé si elle est présente, même si elle est vide (str, list, dict, etc.).
        Retourne None uniquement si la clé est absente ou si la valeur est None.
    """
    value = json_data
    try:
        for key in key_path:
            value = value[key]

        # Retourne la valeur même si elle est vide (str vide, liste vide, etc.)
        return value

    except KeyError:
        # Gestion de l'absence de la clé dans le JSON
        error_message = f"La clé {' -> '.join(key_path)} est absente dans le JSON."
        logger.error(error_message)
        raise KeyError(error_message)

    except ValueError as e:
        # Gestion du cas où la valeur est None
        logger.error(f"Alerte : {str(e)}")
        print(f"Alerte : {str(e)}")
        return None

def get_dataframe_from_json(json_data, key_path):
    """
    Crée un DataFrame à partir d'un dictionnaire JSON en utilisant un chemin de clé.

    Args:
        json_data (dict): Dictionnaire JSON contenant les données.
        key_path (list): Liste des clés pour accéder aux données souhaitées dans le JSON.

    Returns:
        pd.DataFrame: DataFrame contenant les données extraites, ou un DataFrame vide si les données sont absentes.
    """
    # Navigue dans le dictionnaire JSON en suivant le chemin de clé
    data = json_data
    for key in key_path:
        data = data.get(key)
        if data is None:  # La clé n'existe pas
            raise ValueError(f"La clé {' -> '.join(key_path)} est absente dans le JSON.")
        if not data:  # La clé existe mais les données sont vides
            return pd.DataFrame()  # Retourne un DataFrame vide

    # Créer le DataFrame à partir des données extraites
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame([data])

    return df


def create_output_directory(base_path, subfolder):
    """
    Crée un sous-dossier dans le chemin de sortie si ce n'est pas déjà fait.

    Args:
        base_path (str): Le chemin de base du répertoire de sortie.
        subfolder (str): Le nom du sous-dossier à ajouter.

    Returns:
        str: Le chemin complet du répertoire de sortie.
    """
    full_path = os.path.join(base_path, subfolder)
    try:
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            logger.info(f"Répertoire créé: {full_path}")
        else:
            logger.info(f"Répertoire déjà existant: {full_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la création du répertoire {full_path}: {str(e)}")
        raise
    return full_path


""" temporaire """
def get_most_recent_json_file(directory):
    try:
        # Liste tous les fichiers JSON dans le répertoire
        files = [os.path.join(directory, f) for f in os.listdir(directory) if
                 f.endswith('.json') and os.path.isfile(os.path.join(directory, f))]
        if not files:
            return None
        # Trie les fichiers par date de modification (le plus récent en premier)
        files.sort(key=os.path.getmtime, reverse=True)
        #chemin complet du fichier le plus récent
        return files[0]
    except Exception as e:
        print(f"Erreur lors de la récupération du fichier JSON le plus récent: {e}")
        return None

current_dir = os.getcwd()

parent_dir = os.path.dirname(os.path.dirname(current_dir))



