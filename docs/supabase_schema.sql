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

-- Table for storing trained model results (supervised + unsupervised)
create table if not exists public.model_results (
    id uuid primary key default gen_random_uuid(),
    result_type text not null,
    payload jsonb not null,
    updated_at timestamptz not null default now(),
    constraint model_results_result_type_key unique (result_type)
);

alter table public.model_results enable row level security;

create policy "Anon read model_results"
on public.model_results
for select
to anon
using (true);

create policy "Service role full access to model_results"
on public.model_results
for all
to service_role
using (true)
with check (true);
