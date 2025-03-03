docker-compose build
docker-compose run pass_alm_7.0 --r "config_SENSI_REV_09_2024.xlsm" -m "ALIM"
docker-compose run pass_alm_7.0 --r "config_SENSI_REV_09_2024.xlsm" -m "SCENARIO+MOTEUR"