-- Table: public.financial_data

-- DROP TABLE IF EXISTS public.financial_data;

CREATE TABLE IF NOT EXISTS public.financial_data
(
    id integer NOT NULL DEFAULT nextval('financial_data_id_seq'::regclass),
    source_file_id integer,
    company_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    metric_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    period_label character varying(50) COLLATE pg_catalog."default" NOT NULL,
    period_date date,
    fiscal_year integer,
    value numeric(20,4),
    value_text character varying(255) COLLATE pg_catalog."default",
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT financial_data_pkey PRIMARY KEY (id),
    CONSTRAINT financial_data_source_file_id_fkey FOREIGN KEY (source_file_id)
        REFERENCES public.source_files (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.financial_data
    OWNER to llm_user;
-- Index: idx_fd_company

-- DROP INDEX IF EXISTS public.idx_fd_company;

CREATE INDEX IF NOT EXISTS idx_fd_company
    ON public.financial_data USING btree
    (company_name COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_fd_fiscal_year

-- DROP INDEX IF EXISTS public.idx_fd_fiscal_year;

CREATE INDEX IF NOT EXISTS idx_fd_fiscal_year
    ON public.financial_data USING btree
    (fiscal_year ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_fd_metric

-- DROP INDEX IF EXISTS public.idx_fd_metric;

CREATE INDEX IF NOT EXISTS idx_fd_metric
    ON public.financial_data USING btree
    (metric_name COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_fd_period

-- DROP INDEX IF EXISTS public.idx_fd_period;

CREATE INDEX IF NOT EXISTS idx_fd_period
    ON public.financial_data USING btree
    (period_label COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;