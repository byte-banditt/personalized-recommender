from __future__ import annotations
import argparse, asyncio, json, statistics, time
import httpx

async def main():
    p=argparse.ArgumentParser(); p.add_argument('--url',default='http://127.0.0.1:8001/recommend'); p.add_argument('--session-id',type=int,required=True); p.add_argument('--requests',type=int,default=100); p.add_argument('--concurrency',type=int,default=10); a=p.parse_args(); sem=asyncio.Semaphore(a.concurrency); times=[]
    async with httpx.AsyncClient() as client:
        async def one():
            async with sem:
                start=time.perf_counter(); r=await client.post(a.url,json={'session_id':a.session_id}); r.raise_for_status(); times.append((time.perf_counter()-start)*1000)
        await asyncio.gather(*(one() for _ in range(a.requests)))
    times.sort(); print(json.dumps({'requests':a.requests,'concurrency':a.concurrency,'p50_ms':times[len(times)//2],'p99_ms':times[min(len(times)-1,int(len(times)*.99))]}))

if __name__=='__main__': asyncio.run(main())
