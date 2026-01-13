-- Table: public.source_files

-- DROP TABLE IF EXISTS public.source_files;

CREATE TABLE IF NOT EXISTS public.source_files
(
    id integer NOT NULL DEFAULT nextval('source_files_id_seq'::regclass),
    file_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    file_type character varying(50) COLLATE pg_catalog."default" NOT NULL,
    company_name character varying(255) COLLATE pg_catalog."default",
    data_type character varying(100) COLLATE pg_catalog."default",
    loaded_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    raw_json jsonb,
    CONSTRAINT source_files_pkey PRIMARY KEY (id),
    CONSTRAINT source_files_file_name_key UNIQUE (file_name)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.source_files
    OWNER to llm_user;