
import base64
import zlib
import urllib.request
import sys

# The mermaid graph definition
graph = """flowchart TB
  classDef api fill:#e1f5fe,stroke:#01579b,color:black;
  classDef planner fill:#fff3e0,stroke:#e65100,color:black;
  classDef storage fill:#e8f5e9,stroke:#1b5e20,color:black;
  classDef file fill:#f3e5f5,stroke:#4a148c,color:black;

  subgraph UserSpace ["用户空间 (Rank N)"]
    StateDict[("State Dict\n(Model/Optim)")]
    Stateful[("Stateful Object\n(FSDP/DTensor)")]
  end

  subgraph API ["API 层 (Entry Point)"]
    SaveAPI["dcp.save / async_save"]:::api
    LoadAPI["dcp.load"]:::api
  end

  subgraph PlannerLayer ["Planner 层 (规划与协调)"]
    direction TB
    subgraph LocalPlanner ["Local Planner (All Ranks)"]
      L_SavePlan["Save Planner (Write)\nDefaultSavePlanner\n(Gen WriteItem)"]:::planner
      L_LoadPlan["Load Planner (Read)\nDefaultLoadPlanner\n(Gen ReadItem)"]:::planner
    end
    
    subgraph GlobalPlanner ["Coordinator (Rank 0)"]
      G_SavePlan["Global Save (Write)\n(Dedup & Aggregate)"]:::planner
      G_LoadPlan["Global Load (Read)\n(Global View)"]:::planner
    end
  end

  subgraph StorageLayer ["Storage 层 (I/O 执行)"]
    subgraph Writer ["FileSystemWriter"]
      Stager["BlockingAsyncStager\n(CPU Offload/Cache)"]:::storage
      WriteData["write_data\n(Tensor/Bytes)"]:::storage
      Commit["finish\n(Write Metadata)"]:::storage
    end

    subgraph Reader ["FileSystemReader"]
      ReadMeta["read_metadata"]:::storage
      ReadData["read_data\n(Narrow & Copy)"]:::storage
    end
  end

  subgraph FileSystem ["文件系统 (Disk)"]
    MetaFile[(".metadata\n(Index/Schema)")]:::file
    ShardFiles[("__0_0.distcp\n__1_0.distcp\n(Payload)")]:::file
  end

  Stateful -->|"state_dict()"| StateDict
  StateDict --> SaveAPI
  SaveAPI -->|"1. Setup"| L_SavePlan
  L_SavePlan -->|"Local SavePlan\n(WriteItem)"| G_SavePlan
  G_SavePlan -->|"Global SavePlan"| WriteData
  SaveAPI -.->|"Async (Optional)"| Stager
  Stager -.-> WriteData
  WriteData -->|"2. Write Shards"| ShardFiles
  G_SavePlan -->|"Metadata Obj"| Commit
  Commit -->|"3. Commit"| MetaFile

  LoadAPI -.->|"1. Read Index"| ReadMeta
  MetaFile -.-> ReadMeta
  ReadMeta -.->|"Metadata Obj"| L_LoadPlan
  L_LoadPlan -.->|"Local LoadPlan\n(ReadItem)"| G_LoadPlan
  G_LoadPlan -.->|"Final Plan"| ReadData
  ShardFiles -.->|"2. Read Shards"| ReadData
  ReadData -.->|"3. Restore"| StateDict
  StateDict -.->|"load_state_dict()"| Stateful
"""

def generate_mermaid_svg(graph, output_path):
    # Compress using zlib
    compressed_data = zlib.compress(graph.encode('utf-8'), 9)
    # Base64 encode (URL safe)
    base64_string = base64.urlsafe_b64encode(compressed_data).decode('ascii')
    
    # Using kroki.io service
    url = f"https://kroki.io/mermaid/svg/{base64_string}"
    
    print(f"Fetching from {url}...")
    try:
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
            }
        )
        
        with urllib.request.urlopen(req) as response:
            data = response.read()
            with open(output_path, "wb") as f:
                f.write(data)
        print(f"Successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error fetching image: {e}")
        return False

if __name__ == "__main__":
    output_path = "/Users/bytedance/codebase/lucienne999.github.io/public/images/dcp-flowchart.svg"
    generate_mermaid_svg(graph, output_path)
