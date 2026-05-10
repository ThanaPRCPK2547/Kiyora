create extension if not exists "pgcrypto";

create table if not exists public."Kiyora" (
    id uuid primary key default gen_random_uuid(),
    data jsonb not null,
    created_at timestamptz not null default now()
);

alter table public."Kiyora" enable row level security;

create policy "Allow service role full access to Kiyora"
on public."Kiyora"
for all
to service_role
using (true)
with check (true);
