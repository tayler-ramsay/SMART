import os
from pathlib import Path
import json
import logging
from typing import List, Dict, Any
import fnmatch

class CodeCollector:
    def __init__(self, repo_path: str, output_path: str):
        self.repo_path = Path(repo_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def collect_files(self, patterns: List[str]) -> List[Dict[str, Any]]:
        """Collect code files matching patterns"""
        self.logger.info(f"Collecting files from {self.repo_path}")
        collected_files = []
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if any(fnmatch.fnmatch(file, pattern) for pattern in patterns):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            collected_files.append({
                                'path': str(file_path.relative_to(self.repo_path)),
                                'content': content,
                                'language': self._get_language(file)
                            })
                    except Exception as e:
                        self.logger.error(f"Error reading {file_path}: {str(e)}")
        
        self.logger.info(f"Collected {len(collected_files)} files")
        return collected_files
    
    def _get_language(self, filename: str) -> str:
        """Determine language from file extension"""
        ext = filename.lower().split('.')[-1]
        return {
            'ts': 'typescript',
            'tsx': 'typescript',
            'js': 'javascript',
            'jsx': 'javascript'
        }.get(ext, 'unknown')
    
    def save_collected_data(self, files: List[Dict[str, Any]], 
                          filename: str = "collected_code.json"):
        """Save collected files to JSON"""
        output_file = self.output_path / filename
        self.logger.info(f"Saving collected data to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(files, f, indent=2)

