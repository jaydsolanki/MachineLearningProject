DELETE TABLE IF EXISTS ml_models;

CREATE TABLE ml_models(
    id varchar(255) PRIMARY KEY,
    model TEXT
);

Drop table live_news;
CREATE TABLE live_news(
    title TEXT,
    published_at date,
    published_at time,
    description TEXT,
    CONSTRAINT pk_live_news PRIMARY KEY (`published_at`, `title`)
);
