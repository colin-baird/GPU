-- Example PerfettoSQL queries for traces emitted by --trace-file.
-- Load the JSON trace in Perfetto first, then run these queries in the SQL pane.
-- For the full track/event/counter schema, see resources/trace_and_perf_counters.md.

-- Stall breakdown by warp.
select
  thread.name as warp,
  slice.name as state,
  sum(slice.dur) as cycles
from slice
join thread_track on slice.track_id = thread_track.id
join thread on thread_track.utid = thread.utid
where thread.name glob 'Warp *'
group by warp, state
order by warp, cycles desc;

-- Per-warp time spent blocked versus in flight.
select
  thread.name as warp,
  sum(case when slice.name glob 'wait_*' then slice.dur else 0 end) as blocked_cycles,
  sum(case when slice.name not glob 'wait_*' and slice.name != 'retired' then slice.dur else 0 end)
    as in_flight_cycles
from slice
join thread_track on slice.track_id = thread_track.id
join thread on thread_track.utid = thread.utid
where thread.name glob 'Warp *'
group by warp
order by warp;

-- Average hardware counter values across the trace.
select
  counter_track.name,
  avg(counter.value) as avg_value,
  max(counter.value) as peak_value
from counter
join counter_track on counter.track_id = counter_track.id
group by counter_track.name
order by counter_track.name;
