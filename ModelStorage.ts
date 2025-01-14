import fs from 'fs';
import path from 'path';

export class ModelStorage {
  private static generateUniqueFileName(filePath: string): string {
    let newFilePath = filePath;
    let count = 1;

    while (fs.existsSync(newFilePath)) {
      const extname = path.extname(filePath);  
      const basename = path.basename(filePath, extname); 
      newFilePath = `${basename}_${count}${extname}`;
      count++;
    }

    return newFilePath;
  }

  static saveModel(layers: { weights: number[]; biases: number[] }[], filePath: string): void {
    const modelData = layers.map(layer => ({
      weights: layer.weights,
      biases: layer.biases,
    }));

    const uniqueFilePath = this.generateUniqueFileName(filePath);

    fs.writeFileSync(uniqueFilePath, JSON.stringify(modelData), 'utf-8');
    console.log('Model saved to file:', uniqueFilePath);
  }

  static loadModel(filePath: string): { weights: number[]; biases: number[] }[] {
    const modelData = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(modelData);
  }
}
