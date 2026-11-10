"""Passive request-boundary context; counters never identify model pages."""
from evaluator.measurement import memory_delta


def paging_context(before,after,memory_before,memory_after):
    seconds=after['observed_at']-before['observed_at']
    page_size=memory_before['vm']['page_size_bytes']
    result={'environment_window_seconds':seconds,'page_size_bytes':page_size,
        'rate_byte_basis':'VM page-equivalent bytes, not measured disk I/O',
        'swap_used_delta_mib':after['swap_used_mib']-before['swap_used_mib'],
        'memory_delta':memory_delta(memory_before,memory_after),
        'attribution':'system-wide paging; process faults/RSS and allocator bytes do not identify model-page paging'}
    for name in ('swapins','swapouts'):
        pages=after[name]-before[name]
        result[name]={'pages':pages,'pages_per_second':pages/seconds if seconds>0 and pages>=0 else None,
            'bytes_per_second':pages*page_size/seconds if seconds>0 and pages>=0 and page_size is not None else None}
    return result


def summarize_context(rows):
    result={'observations':len(rows),'paging_affected_observations':sum(
        row['environment_assessment']['swap_affected'] for row in rows),
        'all_observations_retained':True,'paging_filter_applied':False,
        'attribution':'global counters do not identify model or process pages'}
    for counter in ('swapins','swapouts'):
        items=[row['paging_context'][counter] for row in rows]
        rates=[item['pages_per_second'] for item in items if item['pages_per_second'] is not None]
        result[counter]={'total_pages':sum(item['pages'] for item in items),
            'maximum_pages_per_second':max(rates) if rates else None}
    snapshots=[row[key] for row in rows for key in ('memory_before','memory_after')]
    result['memory_pressure_levels']=sorted({value['memory_pressure_level_raw'] for value in snapshots
        if value['memory_pressure_level_raw'] is not None})
    for label,container,key in (('maximum_allocator_active_bytes','mlx_allocator','active_bytes'),
            ('maximum_allocator_cache_bytes','mlx_allocator','cache_bytes'),
            ('maximum_process_rss_bytes','process','rss_bytes')):
        values=[value[container][key] for value in snapshots if value.get(container)
            and value[container].get(key) is not None]
        result[label]=max(values) if values else None
    return result
