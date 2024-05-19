"use client"
import { GSP_NO_RETURNED_VALUE } from 'next/dist/lib/constants';
import React, { CSSProperties } from 'react';
import { api } from '~/trpc/react';

import { useCSVReader } from 'react-papaparse';

const styles = {
  csvReader: {
    display: 'flex',
    flexDirection: 'row',
    marginBottom: 10,
  } as CSSProperties,
  browseFile: {
    width: '20%',
  } as CSSProperties,
  acceptedFile: {
    border: '1px solid #ccc',
    height: 45,
    lineHeight: 2.5,
    paddingLeft: 10,
    width: '80%',
  } as CSSProperties,
  remove: {
    borderRadius: 0,
    padding: '0 20px',
  } as CSSProperties,
  progressBarBackgroundColor: {
    backgroundColor: 'red',
  } as CSSProperties,
};



interface movieEnity {
  "id" : number,
  "title" : string,
  "genres" : string,
}








export default function CSVReader() {
  const { CSVReader } = useCSVReader();

  const addMovie =  api.movie.add.useMutation({
    onSuccess: () => {
      console.log("SUCESSFULLY ADDED")
    },
  });
  
  let results: movieEnity[] = []
  function setResults(results: movieEnity[]) {
    results = results;
  }

  async function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function uploadToDB(results){
    console.log("Starting upload")
    console.log("results", results)
    const data = results.data


    for (let index = 1; index < data.length; index++) {
      const movie = {
        id: Number(data[index][0]),
        title: data[index][1],
        genres: data[index][2],
      }
      console.log(movie)
      addMovie.mutate(movie);
      // await delay(500);
      if (index && index % 10000 == 0) {
        await delay(1000);
      }
    }


    // const x = results.data.map(async(entry) => {
    //   // console.log("adding1");
    //   await addMovie.mutateAsync(entry);
    //   // await new Promise((resolve) => setTimeout(resolve, 100));
    // });
    // Promise.all(x)

  }

  return (
    <CSVReader
      onUploadAccepted={(res: any) => {
        console.log('---------------------------');
        console.log(res);
        uploadToDB(res);
        setResults(res);
        console.log('---------------------------');
      }}
    >
      {({
        getRootProps,
        acceptedFile,
        ProgressBar,
        getRemoveFileProps,
      }: any) => (
        <>
          <div style={styles.csvReader}>
            <button type='button' {...getRootProps()} style={styles.browseFile}>
              Browse file
            </button>
            <div style={styles.acceptedFile}>
              {acceptedFile && acceptedFile.name}
            </div>
            <button {...getRemoveFileProps()} style={styles.remove}>
              Remove
            </button>
          </div>
          <ProgressBar style={styles.progressBarBackgroundColor} />
          <button onClick={()=>uploadToDB()}>
          Push to db
      </button>
        </>
      )}

    </CSVReader>
  );
}