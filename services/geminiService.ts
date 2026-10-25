
import { GoogleGenAI, Type } from "@google/genai";
import { Note } from "../types";

const getAIClient = () => {
  return new GoogleGenAI({ apiKey: process.env.API_KEY });
};

export const summarizeNotes = async (notes: Note[]) => {
  const ai = getAIClient();
  const notesText = notes.map(n => `Title: ${n.title}\nContent: ${n.content}`).join('\n---\n');
  
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `以下是我的笔记，请帮我总结我最近关注的领域和知识重点：\n\n${notesText}`,
    config: {
      temperature: 0.7,
    },
  });
  
  // Return an empty string if response.text is undefined
  return response.text || "";
};

export const generateWritingPrompts = async (context: string) => {
  const ai = getAIClient();
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `根据我目前的状态：'${context}'，给我5个适合记录在个人网页上的博客或生活随笔题目。`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            topic: { type: Type.STRING },
            reason: { type: Type.STRING }
          },
          required: ["topic", "reason"]
        }
      }
    }
  });

  const text = response.text;
  if (!text) {
    return [];
  }

  try {
    return JSON.parse(text.trim());
  } catch (e) {
    console.error("Failed to parse AI response as JSON:", e);
    return [];
  }
};
