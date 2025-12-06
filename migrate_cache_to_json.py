"""
Script to migrate existing pickle (.pkl) cache files to JSON format
"""
import os
import pickle
import json
from pathlib import Path

def migrate_cache_files(cache_dir: str = "./chunk_cache"):
    """Convert all .pkl cache files to .json format"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    pkl_files = list(cache_path.glob("*.pkl"))
    
    if not pkl_files:
        print("No .pkl files found to migrate")
        return
    
    print(f"Found {len(pkl_files)} .pkl files to migrate")
    
    migrated = 0
    failed = 0
    
    for pkl_file in pkl_files:
        try:
            # Load pickle data
            with open(pkl_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            chunk_ids = cached_data.get('chunk_ids', [])
            chunks = cached_data.get('chunks', {})
            
            # Convert chunk objects to dicts
            chunks_dict = {}
            for chunk_id, chunk_obj in chunks.items():
                if hasattr(chunk_obj, 'model_dump'):
                    chunks_dict[chunk_id] = chunk_obj.model_dump()
                elif hasattr(chunk_obj, 'dict'):
                    chunks_dict[chunk_id] = chunk_obj.dict()
                else:
                    chunks_dict[chunk_id] = chunk_obj.__dict__
            
            # Determine chunk type from first chunk
            chunk_type = 'base'
            if chunks_dict:
                first_chunk = list(chunks_dict.values())[0]
                if 'semantic_score' in first_chunk:
                    chunk_type = 'semantic'
                elif 'title' in first_chunk and 'propositions' in first_chunk:
                    chunk_type = 'agentic'
                elif 'content' in first_chunk:
                    chunk_type = 'recursive'
            
            # Create JSON file
            json_file = pkl_file.with_suffix('.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'chunk_ids': chunk_ids,
                    'chunks': chunks_dict,
                    'chunk_type': chunk_type
                }, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Migrated: {pkl_file.name} -> {json_file.name} ({chunk_type})")
            migrated += 1
            
            # Optionally delete the .pkl file after successful migration
            # pkl_file.unlink()
            
        except Exception as e:
            print(f"✗ Failed to migrate {pkl_file.name}: {e}")
            failed += 1
    
    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Failed: {failed}")
    print(f"\nNote: Original .pkl files are kept. Delete them manually if migration is successful.")

if __name__ == "__main__":
    migrate_cache_files()
