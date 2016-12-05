DELETE TABLE IF EXISTS ml_models;

CREATE TABLE ml_models(
    id varchar(255) PRIMARY KEY,
    model TEXT
);
