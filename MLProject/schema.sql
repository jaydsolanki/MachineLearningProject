DELETE TABLE IF EXISTS ml_models;

CREATE TABLE ml_models(
    id varchar(255) PRIMARY KEY,
    model TEXT
);

CREATE TABLE live_news(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    published_at datetime,
    description TEXT,
    title TEXT
)
