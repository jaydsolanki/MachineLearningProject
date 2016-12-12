DROP TABLE IF EXISTS ml_models;

CREATE TABLE ml_models(
    id varchar(255) PRIMARY KEY,
    model LONGTEXT
);

DROP TABLE IF EXISTS live_news_classification;
DROP TABLE IF EXISTS live_news;

CREATE TABLE live_news(
    id integer PRIMARY KEY AUTO_INCREMENT,
    title varchar(1000),
    published_date date,
    published_time time,
    description MEDIUMTEXT
);

CREATE TABLE live_news_classification(
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    live_news_id INTEGER,
    category INTEGER,
    score DOUBLE,
    algorithm VARCHAR(25),
    CONSTRAINT fk_live_news_classfication_live_news_id FOREIGN KEY (`live_news_id`) REFERENCES live_news (`id`)
);


DROP TABLE IF EXISTS alchemy_news_classification;
DROP TABLE IF EXISTS alchemy_news;
CREATE TABLE alchemy_news(
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    content VARCHAR(1024),
    category varchar(25)
);

CREATE TABLE alchemy_news_classification(
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    alchemy_news_id INTEGER,
    category INTEGER,
    score DOUBLE,
    algorithm VARCHAR(25),
    CONSTRAINT fk_alchemy_news_classfication_alchemy_news_id FOREIGN KEY (`alchemy_news_id`) REFERENCES alchemy_news (`id`)
);
