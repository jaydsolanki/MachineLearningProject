DROP TABLE IF EXISTS ml_models;

CREATE TABLE ml_models(
    id varchar(255) PRIMARY KEY,
    model LONGTEXT
);

DROP TABLE IF EXISTS live_news;
CREATE TABLE live_news(
    title varchar(1000),
    published_date date,
    published_time time,
    description MEDIUMTEXT,
    CONSTRAINT pk_live_news PRIMARY KEY (`title`, `published_date`)
);
