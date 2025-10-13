-- setup_postgres.sql
-- Run this script to set up PostgreSQL databases for the health companion

-- Connect to PostgreSQL as superuser (postgres)
-- psql -U postgres

-- Create database for user data and history
CREATE DATABASE health_companion;

-- Create database for vector store (or use the same database)
CREATE DATABASE health_companion_vectors;

-- Connect to the user data database
\c health_companion

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Connect to the vector store database
\c health_companion_vectors

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Create a user with password (optional, if not using postgres user)
-- CREATE USER health_app WITH PASSWORD 'your_secure_password';
-- GRANT ALL PRIVILEGES ON DATABASE health_companion TO health_app;
-- GRANT ALL PRIVILEGES ON DATABASE health_companion_vectors TO health_app;

-- You can also use a single database for both:
-- Just create one database and enable vector extension on it
-- Then use the same connection URL for both database_url and postgres_vector_url