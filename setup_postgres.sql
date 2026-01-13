-- PostgreSQL Setup Script for LLM Data Extractor
-- Run this as postgres superuser:
-- psql -U postgres -f setup_postgres.sql

-- Create user
CREATE USER llm_user WITH PASSWORD 'Matrix@2025';

-- Create database
CREATE DATABASE llmdb OWNER llm_user;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE llmdb TO llm_user;

-- Connect to the database and grant schema permissions
\c llmdb
GRANT ALL ON SCHEMA public TO llm_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO llm_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO llm_user;

-- Verify setup
\du llm_user
\l llmdb
